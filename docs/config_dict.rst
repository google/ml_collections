ml_collections.config_dict package
==================================

.. currentmodule:: ml_collections.config_dict

.. automodule:: ml_collections.config_dict

ConfigDict class
----------------
.. autoclass:: ConfigDict
   :members: __init__, is_type_safe, convert_dict, lock, is_locked, unlock,
    get, get_oneway_ref, items, iteritems, keys, iterkeys, values, itervalues,
    eq_as_configdict, to_yaml, to_json, to_json_best_effort, to_dict,
    copy_and_resolve_references, unlocked, ignore_type, get_type, update,
    update_from_flattened_dict

FrozenConfigDict class
----------------------
.. autoclass:: FrozenConfigDict
   :members: __init__

FieldReference class
--------------------
.. autoclass:: FieldReference
   :members: __init__, has_cycle, set, empty, get, get_type, identity, to_int,
    to_float, to_str

Additional Methods
------------------
.. autosummary::
   :toctree: __autosummary

    create
    placeholder
    required_placeholder
    recursive_rename
    CustomJSONEncoder
    JSONDecodeError
    MutabilityError
    RequiredValueError