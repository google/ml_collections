# ML Collections

ML Collections is a library of Python Collections designed for ML use cases.

[![Documentation Status](https://readthedocs.org/projects/ml-collections/badge/?version=latest)](https://ml-collections.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ml-collections.svg)](https://badge.fury.io/py/ml-collections)
[![Build Status](https://github.com/google/ml_collections/workflows/Python%20package/badge.svg)](https://github.com/google/ml_collections/actions?query=workflow%3A%22Python+package%22)

## ConfigDict

The two classes called `ConfigDict` and `FrozenConfigDict` are "dict-like" data
structures with dot access to nested elements. Together, they are supposed to be
used as a main way of expressing configurations of experiments and models.

This document describes example usage of `ConfigDict`, `FrozenConfigDict`,
`FieldReference`.

### Features

*   Dot-based access to fields.
*   Locking mechanism to prevent spelling mistakes.
*   Lazy computation.
*   FrozenConfigDict() class which is immutable and hashable.
*   Type safety.
*   "Did you mean" functionality.
*   Human readable printing (with valid references and cycles), using valid YAML
    format.
*   Fields can be passed as keyword arguments using the `**` operator.
*   There are two exceptions to the strong type-safety of the ConfigDict. `int`
    values can be passed in to fields of type `float`. In such a case, the value
    is type-converted to a `float` before being stored. Similarly, all string
    types (including Unicode strings) can be stored in fields of type `str` or
    `unicode`.

### Basic Usage

```python
from ml_collections import config_dict

cfg = config_dict.ConfigDict()
cfg.float_field = 12.6
cfg.integer_field = 123
cfg.another_integer_field = 234
cfg.nested = config_dict.ConfigDict()
cfg.nested.string_field = 'tom'

print(cfg.integer_field)  # Prints 123.
print(cfg['integer_field'])  # Prints 123 as well.

try:
  cfg.integer_field = 'tom'  # Raises TypeError as this field is an integer.
except TypeError as e:
  print(e)

cfg.float_field = 12  # Works: `Int` types can be assigned to `Float`.
cfg.nested.string_field = u'bob'  # `String` fields can store Unicode strings.

print(cfg)
```

### FrozenConfigDict

A `FrozenConfigDict`is an immutable, hashable type of `ConfigDict`:

```python
from ml_collections import config_dict

initial_dictionary = {
    'int': 1,
    'list': [1, 2],
    'tuple': (1, 2, 3),
    'set': {1, 2, 3, 4},
    'dict_tuple_list': {'tuple_list': ([1, 2], 3)}
}

cfg = config_dict.ConfigDict(initial_dictionary)
frozen_dict = config_dict.FrozenConfigDict(initial_dictionary)

print(frozen_dict.tuple)  # Prints tuple (1, 2, 3)
print(frozen_dict.list)  # Prints tuple (1, 2)
print(frozen_dict.set)  # Prints frozenset {1, 2, 3, 4}
print(frozen_dict.dict_tuple_list.tuple_list[0])  # Prints tuple (1, 2)

frozen_cfg = config_dict.FrozenConfigDict(cfg)
print(frozen_cfg == frozen_dict)  # True
print(hash(frozen_cfg) == hash(frozen_dict))  # True

try:
  frozen_dict.int = 2 # Raises TypeError as FrozenConfigDict is immutable.
except AttributeError as e:
  print(e)

# Converting between `FrozenConfigDict` and `ConfigDict`:
thawed_frozen_cfg = config_dict.ConfigDict(frozen_dict)
print(thawed_frozen_cfg == cfg)  # True
frozen_cfg_to_cfg = frozen_dict.as_configdict()
print(frozen_cfg_to_cfg == cfg)  # True
```

### FieldReferences and placeholders

A `FieldReference` is useful for having multiple fields use the same value. It
can also be used for [lazy computation](#lazy-computation).

You can use `placeholder()` as a shortcut to create a `FieldReference` (field)
with a `None` default value. This is useful if a program uses optional
configuration fields.

```python
from ml_collections import config_dict

placeholder = config_dict.FieldReference(0)
cfg = config_dict.ConfigDict()
cfg.placeholder = placeholder
cfg.optional = config_dict.placeholder(int)
cfg.nested = config_dict.ConfigDict()
cfg.nested.placeholder = placeholder

try:
  cfg.optional = 'tom'  # Raises Type error as this field is an integer.
except TypeError as e:
  print(e)

cfg.optional = 1555  # Works fine.
cfg.placeholder = 1  # Changes the value of both placeholder and
                     # nested.placeholder fields.

print(cfg)
```

Note that the indirection provided by `FieldReference`s will be lost if accessed
through a `ConfigDict`.

```python
from ml_collections import config_dict

placeholder = config_dict.FieldReference(0)
cfg.field1 = placeholder
cfg.field2 = placeholder  # This field will be tied to cfg.field1.
cfg.field3 = cfg.field1  # This will just be an int field initialized to 0.
```

### Lazy computation

Using a `FieldReference` in a standard operation (addition, subtraction,
multiplication, etc...) will return another `FieldReference` that points to the
original's value. You can use `FieldReference.get()` to execute the operations
and get the reference's computed value, and `FieldReference.set()` to change the
original reference's value.

```python
from ml_collections import config_dict

ref = config_dict.FieldReference(1)
print(ref.get())  # Prints 1

add_ten = ref.get() + 10  # ref.get() is an integer and so is add_ten
add_ten_lazy = ref + 10  # add_ten_lazy is a FieldReference - NOT an integer

print(add_ten)  # Prints 11
print(add_ten_lazy.get())  # Prints 11 because ref's value is 1

# Addition is lazily computed for FieldReferences so changing ref will change
# the value that is used to compute add_ten.
ref.set(5)
print(add_ten)  # Prints 11
print(add_ten_lazy.get())  # Prints 15 because ref's value is 5
```

If a `FieldReference` has `None` as its original value, or any operation has an
argument of `None`, then the lazy computation will evaluate to `None`.

We can also use fields in a `ConfigDict` in lazy computation. In this case a
field will only be lazily evaluated if `ConfigDict.get_ref()` is used to get it.

```python
from ml_collections import config_dict

config = config_dict.ConfigDict()
config.reference_field = config_dict.FieldReference(1)
config.integer_field = 2
config.float_field = 2.5

# No lazy evaluatuations because we didn't use get_ref()
config.no_lazy = config.integer_field * config.float_field

# This will lazily evaluate ONLY config.integer_field
config.lazy_integer = config.get_ref('integer_field') * config.float_field

# This will lazily evaluate ONLY config.float_field
config.lazy_float = config.integer_field * config.get_ref('float_field')

# This will lazily evaluate BOTH config.integer_field and config.float_Field
config.lazy_both = (config.get_ref('integer_field') *
                    config.get_ref('float_field'))

config.integer_field = 3
print(config.no_lazy)  # Prints 5.0 - It uses integer_field's original value

print(config.lazy_integer)  # Prints 7.5

config.float_field = 3.5
print(config.lazy_float)  # Prints 7.0
print(config.lazy_both)  # Prints 10.5
```

#### Changing lazily computed values

Lazily computed values in a ConfigDict can be overridden in the same way as
regular values. The reference to the `FieldReference` used for the lazy
computation will be lost and all computations downstream in the reference graph
will use the new value.

```python
from ml_collections import config_dict

config = config_dict.ConfigDict()
config.reference = 1
config.reference_0 = config.get_ref('reference') + 10
config.reference_1 = config.get_ref('reference') + 20
config.reference_1_0 = config.get_ref('reference_1') + 100

print(config.reference)  # Prints 1.
print(config.reference_0)  # Prints 11.
print(config.reference_1)  # Prints 21.
print(config.reference_1_0)  # Prints 121.

config.reference_1 = 30

print(config.reference)  # Prints 1 (unchanged).
print(config.reference_0)  # Prints 11 (unchanged).
print(config.reference_1)  # Prints 30.
print(config.reference_1_0)  # Prints 130.
```

#### Cycles

You cannot create cycles using references. Fortunately
[the only way](#changing-lazily-computed-values) to create a cycle is by
assigning a computed field to one that *is not* the result of computation. This
is forbidden:

```python
from ml_collections import config_dict

config = config_dict.ConfigDict()
config.integer_field = 1
config.bigger_integer_field = config.get_ref('integer_field') + 10

try:
  # Raises a MutabilityError because setting config.integer_field would
  # cause a cycle.
  config.integer_field = config.get_ref('bigger_integer_field') + 2
except config_dict.MutabilityError as e:
  print(e)
```

### Advanced usage

Here are some more advanced examples showing lazy computation with different
operators and data types.

```python
from ml_collections import config_dict

config = config_dict.ConfigDict()
config.float_field = 12.6
config.integer_field = 123
config.list_field = [0, 1, 2]

config.float_multiply_field = config.get_ref('float_field') * 3
print(config.float_multiply_field)  # Prints 37.8

config.float_field = 10.0
print(config.float_multiply_field)  # Prints 30.0

config.longer_list_field = config.get_ref('list_field') + [3, 4, 5]
print(config.longer_list_field)  # Prints [0, 1, 2, 3, 4, 5]

config.list_field = [-1]
print(config.longer_list_field)  # Prints [-1, 3, 4, 5]

# Both operands can be references
config.ref_subtraction = (
    config.get_ref('float_field') - config.get_ref('integer_field'))
print(config.ref_subtraction)  # Prints -113.0

config.integer_field = 10
print(config.ref_subtraction)  # Prints 0.0
```

### Equality checking

You can use `==` and `.eq_as_configdict()` to check equality among `ConfigDict`
and `FrozenConfigDict` objects.

```python
from ml_collections import config_dict

dict_1 = {'list': [1, 2]}
dict_2 = {'list': (1, 2)}
cfg_1 = config_dict.ConfigDict(dict_1)
frozen_cfg_1 = config_dict.FrozenConfigDict(dict_1)
frozen_cfg_2 = config_dict.FrozenConfigDict(dict_2)

# True because FrozenConfigDict converts lists to tuples
print(frozen_cfg_1.items() == frozen_cfg_2.items())
# False because == distinguishes the underlying difference
print(frozen_cfg_1 == frozen_cfg_2)

# False because == distinguishes these types
print(frozen_cfg_1 == cfg_1)
# But eq_as_configdict() treats both as ConfigDict, so these are True:
print(frozen_cfg_1.eq_as_configdict(cfg_1))
print(cfg_1.eq_as_configdict(frozen_cfg_1))
```

### Equality checking with lazy computation

Equality checks see if the computed values are the same. Equality is satisfied
if two sets of computations are different as long as they result in the same
value.

```python
from ml_collections import config_dict

cfg_1 = config_dict.ConfigDict()
cfg_1.a = 1
cfg_1.b = cfg_1.get_ref('a') + 2

cfg_2 = config_dict.ConfigDict()
cfg_2.a = 1
cfg_2.b = cfg_2.get_ref('a') * 3

# True because all computed values are the same
print(cfg_1 == cfg_2)
```

### Locking and copying

Here is an example with `lock()` and `deepcopy()`:

```python
import copy
from ml_collections import config_dict

cfg = config_dict.ConfigDict()
cfg.integer_field = 123

# Locking prohibits the addition and deletion of new fields but allows
# modification of existing values.
cfg.lock()
try:
  cfg.integer_field = 124  # Raises AttributeError and suggests valid field.
except AttributeError as e:
  print(e)
with cfg.unlocked():
  cfg.integer_field = 1555  # Works fine too.

# Get a copy of the config dict.
new_cfg = copy.deepcopy(cfg)
new_cfg.integer_field = -123  # Works fine.

print(cfg)
```

### Dictionary attributes and initialization

```python
from ml_collections import config_dict

referenced_dict = {'inner_float': 3.14}
d = {
    'referenced_dict_1': referenced_dict,
    'referenced_dict_2': referenced_dict,
    'list_containing_dict': [{'key': 'value'}],
}

# We can initialize on a dictionary
cfg = config_dict.ConfigDict(d)

# Reference structure is preserved
print(id(cfg.referenced_dict_1) == id(cfg.referenced_dict_2))  # True

# And the dict attributes have been converted to ConfigDict
print(type(cfg.referenced_dict_1))  # ConfigDict

# However, the initialization does not look inside of lists, so dicts inside
# lists are not converted to ConfigDict
print(type(cfg.list_containing_dict[0]))  # dict
```

### More Examples

For more examples, take a look at
[`ml_collections/config_dict/examples/`](https://github.com/google/ml_collections/tree/master/ml_collections/config_dict/examples)

For examples and gotchas specifically about initializing a ConfigDict, see
[`ml_collections/config_dict/examples/config_dict_initialization.py`](https://github.com/google/ml_collections/blob/master/ml_collections/config_dict/examples/config_dict_initialization.py).

## Config Flags

This library adds flag definitions to `absl.flags` to handle config files. It
does not wrap `absl.flags` so if using any standard flag definitions alongside
config file flags, users must also import `absl.flags`.

Currently, this module adds two new flag types, namely `DEFINE_config_file`
which accepts a path to a Python file that generates a configuration, and
`DEFINE_config_dict` which accepts a configuration directly. Configurations are
dict-like structures (see [ConfigDict](#configdict)) whose nested elements
can be overridden using special command-line flags. See the examples below
for more details.

### Usage

Use `ml_collections.config_flags` alongside `absl.flags`. For
example:

`script.py`:

```python
from absl import app
from absl import flags

from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file('my_config')
_MY_FLAG = flags.DEFINE_integer('my_flag', None)

def main(_):
  print(_CONFIG.value)
  print(_MY_FLAG.value)

if __name__ == '__main__':
  app.run(main)
```

`config.py`:

```python
# Note that this is a valid Python script.
# get_config() can return an arbitrary dict-like object. However, it is advised
# to use ml_collections.config_dict.ConfigDict.
# See ml_collections/config_dict/examples/config_dict_basic.py

from ml_collections import config_dict

def get_config():
  config = config_dict.ConfigDict()
  config.field1 = 1
  config.field2 = 'tom'
  config.nested = config_dict.ConfigDict()
  config.nested.field = 2.23
  config.tuple = (1, 2, 3)
  return config
```

Now, after running:

```bash
python script.py --my_config=config.py \
                 --my_config.field1=8 \
                 --my_config.nested.field=2.1 \
                 --my_config.tuple='(1, 2, (1, 2))'
```

we get:

```
field1: 8
field2: tom
nested:
  field: 2.1
tuple: !!python/tuple
- 1
- 2
- !!python/tuple
  - 1
  - 2
```

Usage of `DEFINE_config_dict` is similar to `DEFINE_config_file`, the main
difference is the configuration is defined in `script.py` instead of in a
separate file.

`script.py`:

```python
from absl import app

from ml_collections import config_dict
from ml_collections import config_flags

config = config_dict.ConfigDict()
config.field1 = 1
config.field2 = 'tom'
config.nested = config_dict.ConfigDict()
config.nested.field = 2.23
config.tuple = (1, 2, 3)

_CONFIG = config_flags.DEFINE_config_dict('my_config', config)

def main(_):
  print(_CONFIG.value)

if __name__ == '__main__':
  app.run()
```

`config_file` flags are compatible with the command-line flag syntax. All the
following options are supported for non-boolean values in configurations:

*   `-(-)config.field=value`
*   `-(-)config.field value`

Options for boolean values are slightly different:

*   `-(-)config.boolean_field`: set boolean value to True.
*   `-(-)noconfig.boolean_field`: set boolean value to False.
*   `-(-)config.boolean_field=value`: `value` is `true`, `false`, `True` or
    `False`.

Note that `-(-)config.boolean_field value` is not supported.

### Parameterising the get_config() function

It's sometimes useful to be able to pass parameters into `get_config`, and
change what is returned based on this configuration. One example is if you are
grid searching over parameters which have a different hierarchical structure -
the flag needs to be present in the resulting ConfigDict. It would be possible
to include the union of all possible leaf values in your ConfigDict,
but this produces a confusing config result as you have to remember which
parameters will actually have an effect and which won't.

A better system is to pass some configuration, indicating which structure of
ConfigDict should be returned. An example is the following config file:

```python
from ml_collections import config_dict

def get_config(config_string):
  possible_structures = {
      'linear': config_dict.ConfigDict({
          'model_constructor': 'snt.Linear',
          'model_config': config_dict.ConfigDict({
              'output_size': 42,
          }),
      'lstm': config_dict.ConfigDict({
          'model_constructor': 'snt.LSTM',
          'model_config': config_dict.ConfigDict({
              'hidden_size': 108,
          })
      })
  }

  return possible_structures[config_string]
```

The value of `config_string` will be anything that is to the right of the first
colon in the config file path, if one exists. If no colon exists, no value is
passed to `get_config` (producing a TypeError if `get_config` expects a value).

The above example can be run like:

```bash
python script.py -- --config=path_to_config.py:linear \
                    --config.model_config.output_size=256
```

or like:

```bash
python script.py -- --config=path_to_config.py:lstm \
                    --config.model_config.hidden_size=512
```

### Additional features

*   Loads any valid python script which defines `get_config()` function
    returning any python object.
*   Automatic locking of the loaded object, if the loaded object defines a
    callable `.lock()` method.
*   Supports command-line overriding of arbitrarily nested values in dict-like
    objects (with key/attribute based getters/setters) of the following types:
    *   `types.IntType` (integer)
    *   `types.FloatType` (float)
    *   `types.BooleanType` (bool)
    *   `types.StringType` (string)
    *   `types.TupleType` (tuple)
    *   `enum.Enum` (enum)
*   Overriding is type safe.
*   Overriding of `TupleType` can be done by passing in the `tuple` as a string
    (see the example in the [Usage](#usage) section).
*   The overriding `tuple` object can be of a different size and have different
    types than the original. Nested tuples are also supported.

## Authors
*   Sergio Gómez Colmenarejo - sergomez@google.com
*   Wojciech Marian Czarnecki - lejlot@google.com
*   Nicholas Watters
*   Mohit Reddy - mohitreddy@google.com
